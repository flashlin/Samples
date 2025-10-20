using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MaxBetReducedAgentsPlayersIdEntityConfiguration : IEntityTypeConfiguration<MaxBetReducedAgentsPlayersIdEntity>
    {
        public void Configure(EntityTypeBuilder<MaxBetReducedAgentsPlayersIdEntity> builder)
        {
            builder.ToTable("MaxBetReducedAgentsPlayersId");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

        }
    }
}
